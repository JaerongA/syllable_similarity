from matplotlib.pylab import psd
from util import save
from util.draw import *
from util.functions import *
from util.spect import *


def read_not_mat(notmat, unit='ms'):
    """ read from .not.mat files generated from uisonganal
    Parameters
    ----------
    notmat : path
        Name of the .not.mat file (path)
    unit : (optional)
        milli-second by default. Convert to seconds when specified

    Returns
    -------
    onsets : array
        time stamp for syllable onset (in ms)
    offsets : array
        time stamp for syllable offset (in ms)
    intervals : array
        temporal interval between syllables (i.e. syllable gaps) (in ms)
    durations : array
        durations of each syllable (in ms)
    syllables : str
        song syllables
    contexts : str
        social context ('U' for undirected and 'D' for directed)
    """
    import scipy.io
    onsets = scipy.io.loadmat(notmat)['onsets'].transpose()[0]  # syllable onset timestamp
    offsets = scipy.io.loadmat(notmat)['offsets'].transpose()[0]  # syllable offset timestamp
    intervals = onsets[1:] - offsets[:-1]  # syllable gap durations (interval)
    durations = offsets - onsets  # duration of each syllable
    syllables = scipy.io.loadmat(notmat)['syllables'][0]  # Load the syllable info
    contexts = notmat.name.split('.')[0].split('_')[-1][
        0].upper()  # extract 'U' (undirected) or 'D' (directed) from the file name
    if contexts not in ['U', 'D']:  # if the file was not tagged with Undir or Dir
        contexts = None

    # units are in ms by default, but convert to second with the argument
    if unit == 'second':
        onsets /= 1E3
        offsets /= 1E3
        intervals /= 1E3
        durations /= 1E3

    return onsets, offsets, intervals, durations, syllables, contexts


def get_note_type(syllables, song_db) -> list:
    """
    Function to determine the category of the syllable
    Parameters
    ----------
    syllables : str
    song_db : db

    Returns
    -------
    type_str : list
    """
    type_str = []
    for syllable in syllables:
        if syllable in song_db.motif:
            type_str.append('M')  # motif
        elif syllable in song_db.calls:
            type_str.append('C')  # call
        elif syllable in song_db.introNotes:
            type_str.append('I')  # intro notes
        else:
            type_str.append(None)  # intro notes
    return type_str


def get_psd_mat(data_path, save_path,
                save_psd=False, update=False, open_folder=False, add_date=False,
                nfft=2 ** 10, fig_ext='.png'):
    from analysis.parameters import freq_range
    import numpy as np
    from scipy.io import wavfile
    import matplotlib.colors as colors
    import matplotlib.gridspec as gridspec

    # Parameters
    note_buffer = 20  # in ms before and after each note
    font_size = 12  # figure font size

    # Read from a file if it already exists
    file_name = data_path / 'PSD.npy'

    if save_psd and not update:
        raise Exception("psd can only be save in an update mode or when the .npy does not exist!, set update to TRUE")

    if update or not file_name.exists():

        # Load files
        files = list(data_path.glob('*.wav'))

        psd_list = []  # store psd vectors for training
        file_list = []  # store files names containing psds
        psd_notes = ''  # concatenate all syllables
        psd_context_list = []  # concatenate syllable contexts

        for file in files:

            notmat_file = file.with_suffix('.wav.not.mat')
            onsets, offsets, intervals, durations, syllables, contexts = read_not_mat(notmat_file, unit='ms')
            sample_rate, data = wavfile.read(file)  # note that the timestamp is in second
            length = data.shape[0] / sample_rate
            timestamp = np.round(np.linspace(0, length, data.shape[0]) * 1E3,
                                 3)  # start from t = 0 in ms, reduce floating precision
            contexts = contexts * len(syllables)
            list_zip = zip(onsets, offsets, syllables, contexts)

            for i, (onset, offset, syllable, context) in enumerate(list_zip):

                # Get spectrogram
                ind, _ = extract_ind(timestamp, [onset - note_buffer, offset + note_buffer])
                extracted_data = data[ind]
                spect, freqbins, timebins = spectrogram(extracted_data, sample_rate, freq_range=freq_range)

                # Get power spectral density
                # nfft = int(round(2 ** 14 / 32000.0 * sample_rate))  # used by Dave Mets

                # Get psd after normalization
                psd_seg = psd(normalize(extracted_data), NFFT=nfft, Fs=sample_rate)  # PSD segment from the time range
                seg_start = int(round(freq_range[0] / (sample_rate / float(nfft))))  # 307
                seg_end = int(round(freq_range[1] / (sample_rate / float(nfft))))  # 8192
                psd_power = normalize(psd_seg[0][seg_start:seg_end])
                psd_freq = psd_seg[1][seg_start:seg_end]

                # Plt & save figure
                if save_psd:
                    # Plot spectrogram & PSD
                    fig = plt.figure(figsize=(3.5, 3))
                    fig_name = "{}, note#{} - {} - {}".format(file.name, i, syllable, context)
                    fig.suptitle(fig_name, y=0.95, fontsize=10)
                    gs = gridspec.GridSpec(6, 3)

                    # Plot spectrogram
                    ax_spect = plt.subplot(gs[1:5, 0:2])
                    ax_spect.pcolormesh(timebins * 1E3, freqbins, spect,  # data
                                        cmap='hot_r',
                                        norm=colors.SymLogNorm(linthresh=0.05,
                                                               linscale=0.03,
                                                               vmin=0.5, vmax=100
                                                               ))

                    remove_right_top(ax_spect)
                    ax_spect.set_ylim(freq_range[0], freq_range[1])
                    ax_spect.set_xlabel('Time (ms)', fontsize=font_size)
                    ax_spect.set_ylabel('Frequency (Hz)', fontsize=font_size)

                    # Plot psd
                    ax_psd = plt.subplot(gs[1:5, 2], sharey=ax_spect)
                    ax_psd.plot(psd_power, psd_freq, 'k')
                    ax_psd.spines['right'].set_visible(False), ax_psd.spines['top'].set_visible(False)
                    # ax_psd.spines['bottom'].set_visible(False)
                    # ax_psd.set_xticks([])  # remove xticks
                    plt.setp(ax_psd.set_yticks([]))
                    # plt.show()

                    # Save figures
                    save_path = save.make_dir(save_path, add_date=add_date)
                    save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext, open_folder=open_folder)
                    plt.close(fig)

                psd_list.append(psd_power)
                file_list.append(file.name)
                psd_notes += syllable
                psd_context_list.append(context)

        # Organize data into a dictionary
        data = {
            'psd_list': psd_list,
            'file_list': file_list,
            'psd_notes': psd_notes,
            'psd_context': psd_context_list,
        }
        # Save results
        np.save(file_name, data)

    else:  # if not update or file already exists
        data = np.load(file_name, allow_pickle=True).item()

    return data['psd_list'], data['file_list'], data['psd_notes'], data['psd_context']


def get_basis_psd(psd_list, notes, song_note=None, num_note_crit_basis=30):
    """
    Get avg psd from the training set (will serve as a basis)
    Parameters
    ----------
    psd_list : list
        List of syllable psds
    notes : str
        String of all syllables
    song_note : str
        String of all syllables
    num_note_crit_basis : int (30 by default)
        Minimum number of notes required to be a basis syllable

    Returns
    -------
    psd_list_basis : list
    note_list_basis : list
    """
    import numpy as np

    psd_dict = {}
    psd_list_basis = []
    note_list_basis = []

    psd_array = np.asarray(psd_list)  # number of syllables x psd (get_basis_psd function accepts array format only)
    unique_note = unique(''.join(sorted(notes)))  # convert note string into a list of unique syllables

    # Remove unidentifiable note (e.g., '0' or 'x')
    if '0' in unique_note:
        unique_note.remove('0')
    if 'x' in unique_note:
        unique_note.remove('x')

    for note in unique_note:
        if note not in song_note: continue
        ind = find_str(notes, note)
        if len(ind) >= num_note_crit_basis:  # number should exceed the  criteria
            note_pow_array = psd_array[ind, :]
            note_pow_avg = note_pow_array.mean(axis=0)
            temp_dict = {note: note_pow_avg}
            psd_list_basis.append(note_pow_avg)
            note_list_basis.append(note)
            psd_dict.update(temp_dict)  # basis
            # plt.plot(psd_dict[note])
            # plt.show()
    return psd_list_basis, note_list_basis